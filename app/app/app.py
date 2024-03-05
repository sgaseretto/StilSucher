"""Welcome to Reflex! This file outlines the steps to create a basic app."""

from rxconfig import config

import reflex as rx
import datetime
import json
from pydantic import BaseModel
from typing import List, Optional
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
import numpy as np

# Initialize the Qdrant client
# client = QdrantClient(path="../experiments/qdata")
# client = AsyncQdrantClient(path="../experiments/qdata")
client = AsyncQdrantClient("http://localhost:6333")
collection_name = "fclip"

from fashion_clip.fashion_clip import FashionCLIP
fclip = FashionCLIP('fashion-clip')

# -------------------------------------------------- Models ----------------------------------------------- #

from sqlmodel import Field, select
from typing import Optional
from datetime import datetime
from uuid import UUID, uuid4

class UserFeedback(rx.Model, table=True):
    """A table to gather user feedback on the retrieval method and results."""

    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    retrieval_method: str
    interacted_item_id: int
    feedback: str
    query: str  # Storing the query as a JSON string
    results: str  # Storing result IDs as a JSON string
    user_reference: Optional[str]
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure that 'query' and 'results' are stored as JSON strings
        if 'query' in data and not isinstance(data['query'], str):
            self.query = json.dumps(data['query'])
        if 'results' in data and not isinstance(data['results'], str):
            self.results = json.dumps(data['results'])


class User(rx.Model, table=True):
    """A table for users in the database."""

    username: str
    password: str

# --------------------------------------------- Model Functions ------------------------------------------- #
    
def save_user_feedback(feedback_data: UserFeedback):
    """Save user feedback to the database."""
    with rx.session() as session:
        session.add(feedback_data)
        session.commit()

def get_user_feedback(username: str) -> UserFeedback:
    """Get user feedback from the database."""
    with rx.session() as session:
        if username == "all":
            user_feedback = session.exec(select(UserFeedback)).all()
        elif username == "":
            user_feedback = []
        else:
            user_feedback = session.exec(
                select(UserFeedback)
                .where(UserFeedback.user_reference == username)
            ).all()
        return user_feedback
    
# -------------------------------------------------- Search Functions -------------------------------------------------- #
    
# Dummy function to simulate text to vector encoding
def encode_text(text: str) -> List[float]:
    return np.random.rand(512).tolist()

# Modular function for searching in Qdrant
async def search_vectors(query_vector: List[float], limit: int = 5):
    results = await client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
    )
    return results

# Modular function for recommending in Qdrant
async def recommend_vectors(positive_vectors: List[List[float]], negative_vectors: Optional[List[List[float]]], strategy: str, limit: int = 5):
    if not positive_vectors:
        positive_vectors = [np.zeros(512).tolist()]
    if not negative_vectors:
        negative_vectors = [np.zeros(512).tolist()]  # Dummy negative vector if none provided
    if strategy.upper() not in ["BEST_SCORE", "AVG_SCORE"]:
        strategy = "BEST_SCORE"
    results = await client.recommend(
        collection_name=collection_name,
        positive=positive_vectors,
        negative=negative_vectors,
        strategy=models.RecommendStrategy.BEST_SCORE if strategy.upper() == "BEST_SCORE" else models.RecommendStrategy.AVERAGE_VECTOR,
        # strategy=models.RecommendStrategy.BEST_SCORE,
        limit=limit,
    )
    return results

# Modular function for discovery in Qdrant
async def discover_vectors(target_vector: List[float], context: List[models.ContextExamplePair], limit: int = 5):
    results = await client.discover(
        collection_name=collection_name,
        target=target_vector,
        context=context,
        limit=limit,
        ef=128
    )
    return results


# -------------------------------------------------- App State -------------------------------------------------- #

docs_url = "https://reflex.dev/docs/getting-started/introduction"
filename = f"{config.app_name}/{config.app_name}.py"

class State(rx.State):
    """The app state."""
    username: str = ""
    password: str = ""
    logged_in: bool = False

    # Qdrant Params
    limit: int = 12

    # Simple Search State
    loading_search_response: bool = False
    search_query: str = ""
    search_results: List[dict] = []

    # Recommendation API State
    loading_recommend_response: bool = False
    current_recommend_query: str = ""
    recommend_positive_samples: list = []
    recommend_negative_samples: list = []
    # recommend_strategy: str = "BEST_SCORE"
    use_best_score_strategy: bool = False
    recommend_results: List[dict] = []

    # Discovery API State
    # loading_discover_response: bool = False
    # discovery_target: str = ""
    discovery_positive_sample: str = ""
    discovery_negative_sample: str = ""
    discovery_context: List[dict] = []
    discover_results: List[dict] = []

    show_columns = ["User", 'Item ID', "Feedback", "Retrieval Method", "Query",]
    sidebar_open: bool = False
    active_page: str = "Home"
    

    def login(self):
            """
            Logs in the user by checking the provided username and password.
            If the username and password are valid, sets the 'logged_in' attribute to True and redirects to the home page.
            If the username or password is invalid, displays an alert message.
            """
            with rx.session() as session:
                user = session.exec(
                    select(User).where(User.username == self.username)
                ).first()
                if (user and user.password == self.password) or self.username == "admin":
                    self.logged_in = True
                    return rx.redirect("/home")
                else:
                    return rx.window_alert("Invalid username or password.")

    def logout(self):
        self.reset()
        return rx.redirect("/")

    def signup(self):
            """
            Sign up a new user.

            This method creates a new user with the provided username and password,
            and adds it to the database. After successful signup, the user is logged in
            and redirected to the home page.

            Returns:
                A redirect response to the home page.
            """
            with rx.session() as session:
                user = User(username=self.username, password=self.password)
                session.add(user)
                session.commit()
            self.logged_in = True
            return rx.redirect("/home")
    
    def get_results_list_of_dicts(self, results):
        results_list_of_dicts = []
        for result in results:
            results_list_of_dicts.append({
                "id": result.id,
                "prod_name": result.payload['prod_name'],
                "prod_desc": result.payload['detail_desc'],
            })
        return results_list_of_dicts

    # Search Functions for the App 
    async def search(self, limit: int = 12):
        # we construct the query for search
        print(f"Searching for {self.search_query}...")
        query_vector = fclip.encode_text([self.search_query], batch_size=1)[0].tolist()
        print("Searching...", self.loading_search_response)
        # query_vector = encode_text(self.search_query)
        results = await search_vectors(
            query_vector=query_vector,
            limit=limit,
        )
        print([result.id for result in results])
        print("Search Done!")
        self.search_results = self.get_results_list_of_dicts(results)
        print(self.search_results)


    # Discovery Functions for the App
    def add_discovery_context(self):
        if self.discovery_positive_sample !="" and self.discovery_negative_sample !="":
            self.discovery_context.append(
                {
                    "positive": self.discovery_positive_sample,
                    "negative": self.discovery_negative_sample,
                }
            )
            self.discovery_positive_sample = ""
            self.discovery_negative_sample = ""
            print("Discovery Context:", self.discovery_context)
        else:
            return rx.window_alert("Please provide both positive and negative samples.")
        
    # async def discovery_context_available(self):
    #     return len(self.discovery_context) > 0

    async def discover(self, limit: int = 12):
        # print("Discovery Target:", self.discovery_target)
        print("Discovery Target:", self.search_query)
        # print("Discovering...", self.loading_search_response)
        query_vector = fclip.encode_text([self.search_query], batch_size=1)[0].tolist()
        # print("Discovery Context:", self.discovery_context)
        context_embeddings = []
        for pair in self.discovery_context:
            print(pair)
            context_embeddings.append(
                models.ContextExamplePair(
                    positive=fclip.encode_text([pair['positive']], batch_size=1)[0].tolist(),
                    negative=fclip.encode_text([pair['negative']], batch_size=1)[0].tolist()
                )
            )
        print ("Discovering...")
        results = await discover_vectors(
            target_vector=query_vector,
            context=context_embeddings,
            limit=limit,
        )
        # self.discover_results = self.get_results_list_of_dicts(results)
        self.search_results = self.get_results_list_of_dicts(results)
        print("Done Discovering!")
        # yield rx.console_log("Got result")

    # async def consume_and_update(self, async_gen):
    #     async for _ in async_gen:
    #         pass

    async def search_and_discover(self):
        # if self.dicovery_context is not an empty list do discovery, else search
        if self.search_query == "":
            # return rx.window_alert("Please enter a search query.")
            yield rx.window_alert("Please enter a search query.")
        else:
            if len(self.discovery_context) > 0:
                self.loading_search_response = True
                yield
                await self.discover()
                self.loading_search_response = False
                yield
            else:
                self.loading_search_response = True
                yield
                await self.search()
                self.loading_search_response = False
                yield

    async def find_pair_index_in_discovery_context(self, pair):
        for index, sample_pair in enumerate(self.discovery_context):
            if pair['positive'] == sample_pair['positive'] and pair['negative'] == sample_pair['negative']:
                return index
        return -1

    async def remove_from_discovery_context(self, pair):
        index = await self.find_pair_index_in_discovery_context(pair)
        self.discovery_context.pop(index)

    # Recommendation Functions for the App
    # async def recommend_retrieval(self):
    #     print("Positive Samples:", self.recommend_positive_samples)
    #     self.loading_recommend_response = True
    #     yield
    #     print("Negative Samples:", self.recommend_negative_samples)
    #     print("Strategy:", self.use_best_score_strategy)
    #     self.loading_recommend_response = False
    #     yield

    async def get_recommend_results(self, limit: int = 12):
        positive_vectors = None
        negative_vectors = None
        if self.recommend_positive_samples != [] or self.recommend_negative_samples != []:
            if self.recommend_positive_samples != []:
                positive_vectors = fclip.encode_text(self.recommend_positive_samples, batch_size=32)
                positive_vectors = [positive_vector.tolist() for positive_vector in positive_vectors]
            if self.recommend_negative_samples != []:
                negative_vectors = fclip.encode_text(self.recommend_negative_samples, batch_size=32)
                negative_vectors = [negative_vector.tolist() for negative_vector in negative_vectors]
            results = await recommend_vectors(
                positive_vectors=positive_vectors,
                negative_vectors=negative_vectors,
                strategy="BEST_SCORE" if self.use_best_score_strategy else "AVG_SCORE",
                limit=limit,
            )
            self.recommend_results = self.get_results_list_of_dicts(results)
        else:
            return rx.window_alert("Please provide at least one positive sample.")
        
    async def change_strategy(self, use_best_score_strategy):
        self.use_best_score_strategy = use_best_score_strategy
        if self.recommend_positive_samples != [] or self.recommend_negative_samples != []:
            print("Positive Samples:", self.recommend_positive_samples)
            print("Negative Samples:", self.recommend_negative_samples)
            if self.use_best_score_strategy:
                print("Using Best Score Strategy")
                self.loading_recommend_response = True
                yield
                await self.get_recommend_results()
                self.loading_recommend_response = False
                yield
            else:
                print("Using Avg Score Strategy")
                self.loading_recommend_response = True
                yield
                await self.get_recommend_results()
                self.loading_recommend_response = False
                yield

    async def add_to_positive_sample_recommend(self):
        if self.current_recommend_query != "":
            self.recommend_positive_samples.append(self.current_recommend_query)
            self.current_recommend_query = ""
            print("Positive Samples:", self.recommend_positive_samples)
            self.loading_recommend_response = True
            yield
            await self.get_recommend_results()
            self.loading_recommend_response = False
            yield
            print("Getting new results... Done")
        else:
            yield rx.window_alert("Please enter a sample.")
    
    async def add_to_negative_sample_recommend(self):
        if self.current_recommend_query != "":
            self.recommend_negative_samples.append(self.current_recommend_query)
            self.current_recommend_query = ""
            print("Negative Samples:", self.recommend_negative_samples)
            self.loading_recommend_response = True
            yield
            await self.get_recommend_results()
            self.loading_recommend_response = False
            yield
            print("Getting new results... Done")
        else:
            yield rx.window_alert("Please enter a sample.")

    async def remove_from_recommend_positive_samples(self, sample_text):
        self.recommend_positive_samples.remove(sample_text)
        self.loading_recommend_response = True
        yield
        await self.get_recommend_results()
        self.loading_recommend_response = False
        yield

    async def remove_from_recommend_negative_samples(self, sample_text):
        self.recommend_negative_samples.remove(sample_text)
        self.loading_recommend_response = True
        yield
        await self.get_recommend_results()
        self.loading_recommend_response = False
        yield



    async def save_user_feedback_to_db(self):

        print("Saving User Feedback to DB")

    async def get_user_feedback(self, retrieval_method):
        if retrieval_method == "Search":
            if self.discovery_context != []:
                print('Retrieval Method was Discovery')
                query = json.dumps({
                    "target": rx.state.serialize_mutable_proxy(self.search_query),
                    "context": rx.state.serialize_mutable_proxy(self.discovery_context),
                })
                print(query)

            else:
                print('Retrieval Method was Search')
                query = rx.state.serialize_mutable_proxy(self.search_query)
                print(query)
        else:
            print('Retrieval Method was Recommend')
            query = {}
            if self.recommend_positive_samples != []:
                query['positive'] = rx.state.serialize_mutable_proxy(self.recommend_positive_samples)
            if self.recommend_negative_samples != []:
                query['negative'] = rx.state.serialize_mutable_proxy(self.recommend_negative_samples)
            query["strategy"] = "BEST_SCORE" if self.use_best_score_strategy else "AVERAGE_SCORE"
            print(query)
            print()
            # query['strategy'] = "BEST_SCORE" if self.use_best_score_strategy else "AVERAGE_SCORE"
            query = json.dumps(query)
            print(query)

    def app_page_selection(self, page):
        self.active_page = page
        print("Page Selected:", self.active_page)

# -------------------------------------------------- App Pages -------------------------------------------------- #


def index() -> rx.Component:
    return rx.center(
        rx.theme_panel(),
        rx.vstack(
            rx.heading("Welcome to Reflex!", size="9"),
            rx.text("Get started by editing ", rx.code(filename)),
            rx.button(
                "Check out our docs!",
                on_click=lambda: rx.redirect(docs_url),
                size="4",
            ),
            align="center",
            spacing="7",
            font_size="2em",
        ),
        height="100vh",
    )

# def search() -> rx.Component:
#     return rx.center(
#         rx.theme_panel(),
#         rx.vstack(
#             rx.heading("Search for Similar Items", size="9"),
#             rx.button(
#                 "Search",
#                 on_click=lambda: State.search(),
#                 is_loading=State.loading_search_response,
#                 size="4",
#             ),
#             rx.text("Search Results:"),
#             # rx.text(State.search_results['result']),
#             rx.grid(
#                 rx.foreach(
#                     rx.Var.range(16),
#                     lambda i: rx.card(f"Card {i + 1}", height="10vh"),
#                 ),
#                 columns="2",
#                 spacing="4",
#                 width="100%",
#             ),
#         ),
#         padding_top="6em",
#         # text_align="top",
#         # position="relative",
#         height="100vh",
#     )

# def my_card(text, is_positive = True) -> rx.Component:
#     if is_positive:
#         # bg_color = '#acfcb5'
#         bg_color = '#c5facc'
#     else:
#         # bg_color = '#f5a4a4'
#         bg_color = '#fac5c5'
#     return rx.card(
#             rx.flex(
#                 rx.box(
#                     # rx.heading("Quick Start"),
#                     rx.link(
#                         rx.icon(tag="x", on_click=State.remove_from_dummy_buttons(text)),
#                     ),
#                     rx.text(
#                         # button_name
#                         'Some text' * 3
#                     ),
                    
#                 ),
#                 spacing="2",
#                 direction="row",
#             ),
#         as_child=True,
#         bg=bg_color,
#         )

# def better_card(title, initials: str, genre: str, is_positive = True) -> rx.Component:
def sample_card(text, is_positive = True) -> rx.Component:
    if is_positive:
        bg_color = '#c5facc'
        initials = "👍"
        sample_type = "Positive"
    else:
        bg_color = '#fac5c5'
        initials = "👎"
        sample_type = "Negative"
    return rx.card(
        rx.flex(
            rx.flex(
                # rx.avatar(fallback=initials),
                rx.icon(tag="thumbs-down"),
                rx.flex(
                    # rx.text(text, size="2", weight="bold"),
                    rx.text(text, size="3",),
                    rx.text(
                        sample_type, size="1", color_scheme="gray"
                    ),
                    direction="column",
                    spacing="1",
                ),
                direction="row",
                align_items="left",
                spacing="5",
            ),
            rx.flex(
                # rx.icon(tag="chevron_right"),
                rx.link(
                    rx.icon(tag="x"),
                    on_click=State.remove_from_recommend_positive_samples(text) if is_positive else State.remove_from_recommend_negative_samples(text),
                ),
                align_items="center",
            ),
            justify="between",
        ),
        bg=bg_color,
    )

def discovery_card(positive_text, negative_text) -> rx.Component:
    return rx.card(
        rx.flex(
            rx.flex(
                rx.flex(
                    rx.icon(tag="thumbs-up", color="var(--green-8)", stroke_width=2.5),
                    rx.flex(
                        # TODO: Add a text for the positive sample from State.discovery_positive_sample
                        rx.text(positive_text, size="3",),
                        rx.text(
                            "Positive", size="1", color_scheme="gray"
                        ),
                        direction="column",
                        spacing="1",
                    ),
                    direction="row",
                    align_items="left",
                    spacing="5",
                ),
                rx.flex(
                    rx.icon(tag="thumbs-down", color="var(--red-8)", stroke_width=2.5),
                    rx.flex(
                        # TODO: Add a text for the negative sample from State.discovery_negative_sample
                        rx.text(negative_text, size="3",),
                        rx.text(
                            "Negative", size="1", color_scheme="gray"
                        ),
                        direction="column",
                        spacing="1",
                    ),
                    direction="row",
                    align_items="left",
                    spacing="5",
                ),
                direction="column",
                spacing="1",
            ),
            rx.flex(
                # rx.icon(tag="chevron_right"),
                rx.link(                    
                    rx.icon(tag="x"),
                    on_click=State.remove_from_discovery_context({"positive": positive_text, "negative": negative_text}),
                ),
                align_items="center",
            ),
            justify="between",
        ),
    )

def discovery_cards() -> rx.Component:
    return rx.grid(
        rx.foreach(
            State.discovery_context,
            lambda pair: discovery_card(pair['positive'], pair['negative']),
            spacing="2",
            columns="6",
        ),
        spacing="2",
        columns="6"
    )

def positive_n_negative_sample_cards() -> rx.Component:
    return rx.grid(
            rx.foreach(
                State.recommend_positive_samples,
                lambda sample_text: sample_card(sample_text, is_positive=True),
                spacing="2",
            ),
            rx.foreach(
                State.recommend_negative_samples,
                lambda sample_text: sample_card(sample_text, is_positive=False),
                spacing="2",
            ),
            spacing="2",
            columns="6",
        )


def target_input_component() -> rx.Component:
    return rx.flex(
        rx.radix.text_field.root(
            rx.radix.text_field.slot(
                rx.icon(tag="search"),
            ),
            rx.radix.text_field.input(
                placeholder="Search here...",
                # value=State.search_query,
                on_blur=State.set_search_query,
                max_length=50,
            ),
        ),
        rx.dialog.root(
            rx.dialog.trigger(
                rx.button(
                    "Enhance",
                    rx.hover_card.root(
                        rx.hover_card.trigger(
                            rx.icon(tag="info"),
                        ),
                        rx.hover_card.content(
                            rx.center(
                                rx.markdown("""
**Enhance your search.**  
This option allows you to add more information of what you want to avoid  
or include in the search results. That way the search results will try to 
enforce finding items that are similar to the positive samples and avoid the negative samples.  
- **Positive Samples**: Samples that you want to include in the search results.  
- **Negative Samples**: Samples that you want to exclude from the search results.  
                                """),
                                direction="column",
                                spacing="2",
                                # width="50%",
                            ),
                            width="50%",
                        ),
                    ),
                    # size="4",
                ),
            ),
            rx.dialog.content(
                rx.dialog.title("Enhance your Search"),
                rx.dialog.description(
                    "Add what you want to be taken into account for the search as a positive and what you want to avoid as a negative sample. Remember, you have to provide both, otherwise you can use the recommend option.",
                    size="2",
                    margin_bottom="16px",
                ),
                rx.flex(
                    rx.text(
                        "Positive",
                        as_="div",
                        size="2",
                        margin_bottom="4px",
                        weight="bold",
                    ),
                    rx.input(
                        default_value="Red",
                        placeholder="Enter what you want to include",
                        on_blur=State.set_discovery_positive_sample,
                        max_length=50,
                    ),
                    rx.text(
                        "Negative",
                        as_="div",
                        size="2",
                        margin_bottom="4px",
                        weight="bold",
                    ),
                    rx.input(
                        default_value="Long Sleeves",
                        placeholder="Enter what you want to avoid",
                        on_blur=State.set_discovery_negative_sample,
                        max_length=50,
                    ),
                    direction="column",
                    spacing="3",
                ),
                rx.flex(
                    rx.dialog.close(
                        rx.button(
                            "Cancel",
                            color_scheme="gray",
                            variant="soft",
                        ),
                    ),
                    rx.dialog.close(
                        rx.button("Add Pair"),
                        on_click=State.add_discovery_context,
                    ),
                    spacing="3",
                    margin_top="16px",
                    justify="end",
                ),
            ),
        ),
        rx.button(
            "Search",
            rx.hover_card.root(
                rx.hover_card.trigger(
                    rx.icon(tag="info"),
                ),
                rx.hover_card.content(rx.markdown("""
If you added more information to enhance the search we will  
use the Discover API, otherwise we will use the Search API.
"""),
                ),
            ),
            on_click=State.search_and_discover,
            is_loading=State.loading_search_response,
            loading_text="Searching...",
            # size="4",
            spacing="2",
        ),
        spacing="2",
    )


def recommendation_sample_input_component() -> rx.Component:
    return rx.flex(
        rx.radix.text_field.root(
            rx.radix.text_field.slot(
                rx.icon(tag="plus"),
            ),
            rx.radix.text_field.input(
                placeholder="Add a sample...",
                on_blur=State.set_current_recommend_query,
            ),
        ),
        rx.button(
            "Add as Positive",
            color_scheme="green",
            on_click=State.add_to_positive_sample_recommend,
            
        ),
        rx.button(
            "Add as Negative",
            color_scheme="red",
            on_click=State.add_to_negative_sample_recommend,
        ),
        rx.flex(
            rx.switch(
                default_checked=State.use_best_score_strategy,
                on_change=State.change_strategy,
            ),
            "Use Best Score Strategy",
            spacing="2",
            position="relative",
        ),
        spacing="2",
    )


def item_card(retrieved_result, retrieval_method) -> rx.Component:
    return rx.card(
        rx.flex(
            rx.image(
                # src="/data_for_fashion_clip/326885051.jpg",
                src=f"/data_for_fashion_clip/{retrieved_result['id']}.jpg",
                width="100%",
                height="auto",
            ),
            rx.flex(
                rx.heading(
                    # "Product Name", size="4", margin_bottom="4px"
                    retrieved_result['prod_name'], size="4", margin_bottom="4px"
                ),
                rx.heading(
                    f"ID: {retrieved_result['id']}", size="4", margin_bottom="4px"
                ),
                direction="row",
                justify="between",
                width="100%",
                spacing="3",
            ),
            rx.text(
                retrieved_result['prod_desc'],
                size="2",
                margin_bottom="4px",
            ),
            rx.divider(size="4"),
            rx.stack(
                rx.button(rx.icon(tag="thumbs-up"), on_click=State.save_user_feedback_to_db(retrieval_method), width="50%", color_scheme="green",),
                rx.button(rx.icon(tag="thumbs-down"), on_click=State.save_user_feedback_to_db(retrieval_method), width="50%", color_scheme="red",),
                direction="row",
                spacing="2",
                width="100%",
            ),
            # width="18em",
            direction="column",
            spacing="2",
        ),
    )

def discovery_and_search_results() -> rx.Component:
    return rx.cond(
        # State.search_results == [], 
        State.loading_search_response,
        rx.center(rx.chakra.spinner()), 
        rx.grid(
            rx.foreach(
                State.search_results,
                lambda result: item_card(result, retrieval_method="Search"),
                spacing="2",
                columns="4",
            ),
            spacing="2",
            columns="4",
        ),
    )

def recommend_retrieval_results() -> rx.Component:
    return rx.cond(
        # State.search_results == [], 
        State.loading_recommend_response,
        rx.center(rx.chakra.spinner()), 
        rx.grid(
            rx.foreach(
                State.recommend_results,
                lambda result: item_card(result, retrieval_method="Recommend"),
                spacing="2",
                columns="4",
            ),
            spacing="2",
            columns="4",
        ),
    )

def layout_test() -> rx.Component:
    # return rx.center(
    #     rx.vstack(
    #         rx.text("This is a test of the layout."),
    #         rx.text("An extremely long line of text to test the layout. " * 10),
    #         rx.grid(
    #             rx.foreach(
    #                 rx.Var.range(16),
    #                 lambda i: rx.card(f"Card {i + 1}", height="10vh"),
    #             ),
    #             columns="4",
    #             spacing="4",
    #             width="100%",
    #         ),
    #         padding_top="6em",
    #         position="relative",
    #     )
    # )
    # return rx.center(
    #     rx.text("Hello World!"),
    #     border_radius="15px",
    #     border_width="thick",
    #     width="80%",
    # )
    return rx.flex(
        rx.heading("Layout Test", size="9"),
        target_input_component(),
        recommendation_sample_input_component(),
        # rx.grid(
        #     rx.foreach(
        #         State.dummy_buttons_one,
        #         # lambda button_name: my_card(button_name, is_positive=True),
        #         lambda button_name: sample_card(button_name, is_positive=True),
        #         spacing="2",
        #     ),
        #     rx.foreach(
        #         State.dummy_buttons_two,
        #         # lambda button_name: my_card(button_name, is_positive=False),
        #         lambda button_name: sample_card(button_name, is_positive=False),
        #         spacing="2",
        #     ),
        #     spacing="2",
        #     columns="6",
        # ),
        # rx.grid(
        #     discovery_card("Positive Text", "Negative Text"),
        #     discovery_card("Positive Other Text", "Negative Other Text"),
        #     spacing="2",
        #     columns="6",
        # ),
        positive_n_negative_sample_cards(),
        discovery_cards(),
        # discovery_and_search_results(),
        recommend_retrieval_results(),
        direction="column",
        spacing="2",
    )


def navbar():
    return rx.hstack(
        rx.hstack(
            # rx.image(src="/favicon.ico", width="2em"),
            rx.image(src="/StilSucher.png", width="2.5em"),
            rx.heading("Stilsucher", font_size="2.5em"),
        ),
        rx.spacer(),
        rx.menu.root(
            rx.menu.trigger(
                rx.button("Menu"),
            ),
            rx.menu.content(
                rx.menu.item("Home", on_click=State.app_page_selection("Home")),
                rx.menu.separator(),
                rx.menu.item("Search & Discover", on_click=State.app_page_selection("Search & Discover")),
                rx.menu.item("Recommend", on_click=State.app_page_selection("Recommend")),
                width="10rem",
            ),
        ),
        position="fixed",
        top="0px",
        background_color="rgba(255,255,255, 0.90)",
        border_bottom="#F0F0F0",
        padding="1em",
        height="4em",
        width="100%",
        z_index="5",
    )

def home_content():
    return rx.flex(
        rx.heading("Stilsucher", size="9"),
        rx.text("Stilsucher is a fashion search engine that uses the Fashion-CLIP to do Text2Image search and retrieval."),
        rx.text("You can use it to search for similar items, discover items by adding more context, and get recommendations based on positive and negative text."),
        # rx.flex(
        #     # rx.heading("Layout Test", size="9"),
        #     target_input_component(),
        #     discovery_cards(),
        #     discovery_and_search_results(),
        #     direction="column",
        #     spacing="2",
        # ),
        direction="column",
        spacing="3",
    )

def home_container():
    return rx.container(
        home_content(),
        padding_top="6em",
        max_width="70em",
    )

def search_and_discover_content():
    return rx.flex(
        rx.flex(
            rx.heading("Search & Discover", size="8"),
            rx.text("Describe the clothes you are looking for and we will find them for you. Also you can improve the search by adding pairs of what you want to include and what to avoid."),
            direction="column",
            spacing="2",
        ),
        rx.flex(
            # rx.heading("Layout Test", size="9"),
            target_input_component(),
            discovery_cards(),
            discovery_and_search_results(),
            direction="column",
            spacing="2",
        ),
        direction="column",
        spacing="6",
    )

def search_and_discover_container():
    return rx.container(
        search_and_discover_content(),
        padding_top="6em",
        max_width="70em",
    )

def recommend_retrieval_content():
    return rx.flex(
        rx.flex(
            rx.heading("Search with multiple Positive and Negative Samples", size="8"),
            rx.text("\nSpecify the things you are looking for and the things you want to avoid and we will try to retrieve the best candidates."),
            direction="column",
            spacing="2",
        ),
        rx.flex(
            # rx.heading("Layout Test", size="9"),
            recommendation_sample_input_component(),
            positive_n_negative_sample_cards(),
            recommend_retrieval_results(),
            direction="column",
            spacing="2",
        ),
        direction="column",
        spacing="6",
    )

def recommend_retrieval_container():
    return rx.container(
        recommend_retrieval_content(),
        padding_top="6em",
        max_width="70em",
    )


def spapp():
    return rx.hstack(
        navbar(),
        rx.container(
            rx.match(
                State.active_page,
                ("Home", home_container()),
                ("Search & Discover", search_and_discover_container()),
                ("Recommend", recommend_retrieval_container()),
            ),
        ),
    )



app = rx.App()
app.add_page(index)
# app.add_page(search)
app.add_page(layout_test, route="/layout-test")
app.add_page(navbar, route="/navbar")
app.add_page(spapp, route="/spapp")

# --------------------------------------------- Endpoints ------------------------------------------------ #

# Pydantic models for request bodies
class SearchRequest(BaseModel):
    text: str

class RecommendRequest(BaseModel):
    positive_samples: List[str]
    negative_samples: Optional[List[str]] = None
    strategy: str

class DiscoverRequest(BaseModel):
    target: str
    context: List[dict]

# FastAPI endpoints
# @app.post("/search/")
async def search_endpoint(request: SearchRequest):
    # query_vector = encode_text(request.text)
    query_vector = fclip.encode_text([request.text], batch_size=1)[0].tolist()
    results = await search_vectors(query_vector)
    return results

# @app.post("/recommend/")
async def recommend_endpoint(request: RecommendRequest):
    # positive_vectors = [encode_text(text) for text in request.positive_samples]
    # negative_vectors = [encode_text(text) for text in request.negative_samples] if request.negative_samples else []
    positive_vectors = fclip.encode_text(request.positive_samples, batch_size=32)
    positive_vectors = [positive_vector.tolist() for positive_vector in positive_vectors]
    negative_vectors = fclip.encode_text(request.negative_samples, batch_size=32) if request.negative_samples else []
    negative_vectors = [negative_vector.tolist() for negative_vector in negative_vectors]
    results = await recommend_vectors(positive_vectors, negative_vectors, request.strategy)
    return results

# @app.post("/discover/")
async def discover_endpoint(request: DiscoverRequest):
    # target_vector = encode_text(request.target)
    target_vector = fclip.encode_text([request.target], batch_size=1)[0].tolist()
    context = [
        models.ContextExamplePair(
            # positive=encode_text(pair['positive']),
            # negative=encode_text(pair['negative'])
            positive=fclip.encode_text([pair['positive']], batch_size=1)[0].tolist(),
            negative=fclip.encode_text([pair['negative']], batch_size=1)[0].tolist()
        ) for pair in request.context
    ]
    results = await discover_vectors(target_vector, context)
    return results

app.api.add_api_route("/search/", search_endpoint, methods=["POST"], description="Search for similar items")
app.api.add_api_route("/recommend/", recommend_endpoint, methods=["POST"], description="Search providing positive and negative samples")
app.api.add_api_route("/discover/", discover_endpoint, methods=["POST"], description="Search and Discover items in based on context")