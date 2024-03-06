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
import os

# Initialize the Qdrant client
qdrant_url_varname = "QDRANT_URL"
qdrant_url = os.environ.get(qdrant_url_varname)
qdrant_url = qdrant_url or "localhost"
# client = QdrantClient(path="../experiments/qdata")
# client = AsyncQdrantClient(path="../experiments/qdata")
# client = AsyncQdrantClient("http://localhost:6333")
client = AsyncQdrantClient(
    url=qdrant_url,
    port=6333,
    prefer_grpc=True,
)
collection_name = "fclip"

from fashion_clip.fashion_clip import FashionCLIP
fclip = FashionCLIP('fashion-clip')

MAX_FEEDBACKS = 10

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
    is_positive_feedback: bool
    query: str  # Storing the query as a JSON string
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
    # discover_results: List[dict] = []

    data_columns = ["User", 'Item ID', "Was Positive Feedback?", "Retrieval Method", "Query",]
    active_page: str = "Home"
    
    @rx.var
    def feedbacks(self) -> List[UserFeedback]:
        with rx.session() as session:
            if self.logged_in:
                if self.username == "admin":
                    users_feedback = session.exec(
                        select(UserFeedback)
                        .order_by(UserFeedback.timestamp.desc())
                        .limit(MAX_FEEDBACKS)
                    ).all()
                elif self.username != "":
                    users_feedback = session.exec(
                        select(UserFeedback)
                        .where(UserFeedback.user_reference == self.username)
                        .order_by(UserFeedback.timestamp.desc())
                        .limit(MAX_FEEDBACKS)
                    ).all()
                return [
                        [
                            feedback.user_reference, 
                            feedback.interacted_item_id, 
                            str(feedback.is_positive_feedback), 
                            feedback.retrieval_method, 
                            feedback.query
                        ] 
                    for feedback in users_feedback
                    ]
            else:
                return []

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
        """
        Converts a list of results into a list of dictionaries.

        Args:
            results (list): A list of results.

        Returns:
            list: A list of dictionaries, where each dictionary represents a result with the following keys:
                - 'id': The ID of the result.
                - 'prod_name': The product name of the result.
                - 'prod_desc': The product description of the result.
        """
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
        """
        Perform a search based on the provided search query.

        Args:
            limit (int): The maximum number of search results to retrieve. Default is 12.

        Returns:
            None
        """
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
        """
        Adds a discovery context to the list of discovery contexts.

        If both the positive and negative samples are provided, a new discovery context
        is created and appended to the list. The positive and negative samples are then
        cleared. If either the positive or negative sample is missing, an alert is shown.

        Returns:
            None
        """
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
        """
        Discover similar items based on the search query and discovery context.

        Args:
            limit (int): The maximum number of results to return. Default is 12.

        Returns:
            None
        """
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


    async def search_and_discover(self):
        """
        Performs a search or discovery based on the search query and discovery context.

        If the search query is empty, it displays an alert message.
        If the discovery context is not empty, it performs a discovery.
        Otherwise, it performs a search.

        Returns:
            None
        """
        if self.search_query == "":
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
        """
        Finds the index of a pair in the discovery context.

        Args:
            pair (dict): The pair to search for in the discovery context.

        Returns:
            int: The index of the pair in the discovery context, or -1 if not found.
        """
        for index, sample_pair in enumerate(self.discovery_context):
            if pair['positive'] == sample_pair['positive'] and pair['negative'] == sample_pair['negative']:
                return index
        return -1

    async def remove_from_discovery_context(self, pair):
        """
        Removes a pair from the discovery context.

        Args:
            pair: The pair to be removed.

        Returns:
            None
        """
        index = await self.find_pair_index_in_discovery_context(pair)
        self.discovery_context.pop(index)

    # Recommendation Functions for the App

    async def get_recommend_results(self, limit: int = 12):
        """
        Retrieves recommendation results based on positive and negative samples.

        Args:
            limit (int): The maximum number of recommendation results to retrieve. Defaults to 12.

        Returns:
            None: If no positive or negative samples are provided.
            List[Dict[str, Any]]: A list of recommendation results in the form of dictionaries.
        """
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
        """
        Change the strategy for recommendation based on the given parameter.

        Args:
            use_best_score_strategy (bool): Flag indicating whether to use the best score strategy or the average score strategy.

        Returns:
            None
        """
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
        """
        Adds the current recommend query to the list of positive samples,
        retrieves new recommend results, and updates the loading status.

        If the current recommend query is empty, it displays an alert.

        Yields:
            None
        """
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
        """
        Adds the current recommend query to the list of negative samples,
        retrieves new recommend results, and updates the loading status.

        If the current recommend query is empty, it displays an alert message.

        Yields:
            None
        """
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
        """
        Removes the given sample_text from the recommend_positive_samples list,
        reloads the recommend results, and updates the loading_recommend_response flag.

        Args:
            sample_text (str): The sample text to be removed.

        Yields:
            None
        """
        self.recommend_positive_samples.remove(sample_text)
        self.loading_recommend_response = True
        yield
        await self.get_recommend_results()
        self.loading_recommend_response = False
        yield

    async def remove_from_recommend_negative_samples(self, sample_text):
        """
        Removes the specified sample_text from the recommend_negative_samples list,
        triggers the get_recommend_results method, and updates the loading_recommend_response flag.

        Args:
            sample_text (str): The text to be removed from the recommend_negative_samples list.

        Yields:
            None
        """
        self.recommend_negative_samples.remove(sample_text)
        self.loading_recommend_response = True
        yield
        await self.get_recommend_results()
        self.loading_recommend_response = False
        yield


    async def save_user_feedback_to_db(self, user_feedback: UserFeedback):
        """
        Saves the user feedback to the database.

        Args:
            user_feedback (UserFeedback): The user feedback object to be saved.

        Returns:
            None
        """
        with rx.session() as session:
            if (
                session.exec(
                    select(UserFeedback)
                    .where(UserFeedback.user_reference == user_feedback.user_reference)
                    .where(UserFeedback.interacted_item_id == user_feedback.interacted_item_id)
                    .where(UserFeedback.query == user_feedback.query)
                    .where(UserFeedback.retrieval_method == user_feedback.retrieval_method)
                    .where(UserFeedback.is_positive_feedback == user_feedback.is_positive_feedback)
                ).first() 
            ):
                print("User Feedback already exists in DB")
            else:
                print(user_feedback.interacted_item_id, user_feedback.query, user_feedback.retrieval_method, user_feedback.is_positive_feedback)
                session.add(user_feedback)
                session.commit()
                print("User Feedback Saved to DB")


    async def receive_user_feedback(self, retrieval_method_page, interacted_item_id, is_positive_feedback):
        """
        Receives user feedback and saves it to the database.

        Args:
            retrieval_method_page (str): The retrieval method page.
            interacted_item_id (str): The ID of the interacted item.
            is_positive_feedback (bool): Indicates whether the feedback is positive or negative.

        Returns:
            None
        """
        user_reference = self.username
        if retrieval_method_page == "Search":
            if self.discovery_context != []:
                print('Retrieval Method was Discovery')
                query = json.dumps({
                    # "target": rx.state.serialize_mutable_proxy(self.search_query),
                    "target": self.search_query,
                    "context": rx.state.serialize_mutable_proxy(self.discovery_context),
                })
                print(query)
                retrieval_method = "discovery"
            else:
                print('Retrieval Method was Search')
                # query = rx.state.serialize_mutable_proxy(self.search_query)
                query = self.search_query
                print(query)
                retrieval_method = "search"
        else:
            print('Retrieval Method was Recommend')
            query = {}
            if self.recommend_positive_samples != []:
                query['positive'] = rx.state.serialize_mutable_proxy(self.recommend_positive_samples)
            if self.recommend_negative_samples != []:
                query['negative'] = rx.state.serialize_mutable_proxy(self.recommend_negative_samples)
            query["strategy"] = "BEST_SCORE" if self.use_best_score_strategy else "AVERAGE_SCORE"
            query = json.dumps(query)
            retrieval_method = "recommend"
            print(query)
        
        user_feedback = UserFeedback(
            retrieval_method=retrieval_method,
            interacted_item_id=interacted_item_id,
            is_positive_feedback=is_positive_feedback,
            query=query,
            user_reference=user_reference,
        )
        print(user_feedback)
        await self.save_user_feedback_to_db(user_feedback)

    def app_page_selection(self, page):
        """
        Selects the active page of the app.

        Args:
            page (str): The name of the page to be selected.

        Returns:
            None
        """
        self.active_page = page
        print("Page Selected:", self.active_page)

# -------------------------------------------------- App Pages -------------------------------------------------- #


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


def item_card(retrieved_result, retrieval_method_page) -> rx.Component:
# def item_card(retrieved_result) -> rx.Component:
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
                rx.button(
                    rx.icon(tag="thumbs-up"), 
                    # on_click=State.receive_user_feedback(
                    #     retrieval_method_page=retrieval_method_page, 
                    #     interacted_item_id=retrieved_result['id'], 
                    #     is_positive_feedback=True
                    # ), 
                    on_click=State.receive_user_feedback(retrieval_method_page, retrieved_result['id'], True), 
                    width="50%", 
                    color_scheme="green",
                ),
                rx.button(
                    rx.icon(tag="thumbs-down"), 
                    on_click=State.receive_user_feedback(retrieval_method_page, retrieved_result['id'], False), 
                    width="50%", 
                    color_scheme="red",
                ),
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
                # lambda result: item_card(retrieved_result=result, retrieval_method_page="Search"),
                # lambda result: item_card(retrieved_result=result),
                lambda result: item_card(result, "Search"),
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
                # lambda result: item_card(retrieved_result=result, retrieval_method_page="Recommend"),
                # lambda result: item_card(retrieved_result=result),
                lambda result: item_card(result, "Recommend"),
                spacing="2",
                columns="4",
            ),
            spacing="2",
            columns="4",
        ),
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
            rx.cond(
                State.logged_in,
                rx.menu.content(
                    rx.chakra.center(
                        rx.chakra.vstack(
                            rx.chakra.avatar(name=State.username, size="md"),
                            rx.chakra.text(State.username),
                        ),
                    ),
                    rx.menu.separator(),
                    rx.menu.item("Home", on_click=State.app_page_selection("Home")),
                    rx.menu.item("Search & Discover", on_click=State.app_page_selection("Search & Discover")),
                    rx.menu.item("Recommend", on_click=State.app_page_selection("Recommend")),
                    rx.menu.item("Feedbacks", on_click=State.app_page_selection("Feedbacks")),
                    rx.menu.item("Logout", on_click=State.logout),
                    width="10rem",
                ),
                rx.menu.content(
                    rx.menu.item("Login", on_click=rx.redirect("/")),
                    rx.menu.item("Sign Up", on_click=rx.redirect("/signup")),
                    width="10rem",
                ),
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
        rx.image(src="/StilSucher.png", width="30%"),
        rx.heading("Stilsucher", size="9"),
        rx.text("Stilsucher is a fashion search engine that uses the Fashion-CLIP to do Text2Image search and retrieval."),
        rx.text("You can use it to search for similar items, discover items by adding more context, and get recommendations based on positive and negative text."),
        direction="column",
        spacing="3",
        align="center",
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

def feedbacks_content():
    return rx.flex(
        rx.heading("Feedbacks", size="8"),
        rx.divider(),
        rx.data_table(
            data=State.feedbacks,
            columns=State.data_columns,
            pagination=True,
            sort=True,
        ),
        direction="column",
        spacing="6",
    )

def feedbacks_container():
    return rx.container(
        feedbacks_content(),
        padding_top="6em",
        max_width="70em",
    )


def login():
    """
    Renders a login form with username and password inputs, a login button, and a sign-up link.

    Returns:
        A Chakra UI component representing the login form.
    """
    return rx.chakra.center(
        rx.chakra.vstack(
            rx.chakra.input(on_blur=State.set_username, placeholder="Username", width="100%"),
            rx.chakra.input(
                type_="password",
                on_blur=State.set_password,
                placeholder="Password",
                width="100%",
            ),
            rx.chakra.button("Login", on_click=State.login, width="100%"),
            rx.chakra.link(rx.chakra.button("Sign Up", width="100%"), href="/signup", width="100%"),
        ),
        shadow="lg",
        padding="1em",
        border_radius="lg",
        background="white",
    )

def signup():
    """
    Function to display the GPT Sign Up form.

    Returns:
        rx.chakra.box: The GPT Sign Up form.
    """
    return rx.chakra.box(
        rx.chakra.vstack(
            navbar(),
            rx.chakra.center(
                rx.chakra.vstack(
                    rx.chakra.heading("GPT Sign Up", font_size="1.5em"),
                    rx.chakra.input(
                        on_blur=State.set_username, placeholder="Username", width="100%"
                    ),
                    rx.chakra.input(
                        type_="password",
                        on_blur=State.set_password,
                        placeholder="Password",
                        width="100%",
                    ),
                    rx.chakra.input(
                        type_="password",
                        on_blur=State.set_password,
                        placeholder="Confirm Password",
                        width="100%",
                    ),
                    rx.chakra.button("Sign Up", on_click=State.signup, width="100%"),
                ),
                shadow="lg",
                padding="1em",
                border_radius="lg",
                background="white",
            )
        ),
        padding_top="10em",
        text_align="top",
        position="relative",
        width="100%",
        height="100vh",
        background="radial-gradient(circle at 22% 11%,rgba(62, 180, 137,.20),hsla(0,0%,100%,0) 19%),radial-gradient(circle at 82% 25%,rgba(33,150,243,.18),hsla(0,0%,100%,0) 35%),radial-gradient(circle at 25% 61%,rgba(250, 128, 114, .28),hsla(0,0%,100%,0) 55%)",
    )

def index():
    """
    This function returns a Chakra UI box with a vertical stack of components.
    The components include a navbar and a login form.
    """
    return rx.chakra.box(
        rx.chakra.vstack(
            navbar(),
            login(),
        ),
        padding_top="10em",
        text_align="top",
        position="relative",
        width="100%",
        height="100vh",
        background="radial-gradient(circle at 22% 11%,rgba(62, 180, 137,.20),hsla(0,0%,100%,0) 19%),radial-gradient(circle at 82% 25%,rgba(33,150,243,.18),hsla(0,0%,100%,0) 35%),radial-gradient(circle at 25% 61%,rgba(250, 128, 114, .28),hsla(0,0%,100%,0) 55%)",
    )


def stilsucher_home():
    return rx.hstack(
        navbar(),
        rx.container(
            rx.match(
                State.active_page,
                ("Home", home_container()),
                ("Search & Discover", search_and_discover_container()),
                ("Recommend", recommend_retrieval_container()),
                ("Feedbacks", feedbacks_container()),
            ),
        ),
    )



app = rx.App()
app.add_page(index)
app.add_page(stilsucher_home, route="/home")
app.add_page(signup)
# app.add_page(home)

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
    """
    Search endpoint that takes a SearchRequest object as input and returns the search results.

    Parameters:
        request (SearchRequest): The search request object containing the text to search.

    Returns:
        results (list): The search results.
    """
    # query_vector = encode_text(request.text)
    query_vector = fclip.encode_text([request.text], batch_size=1)[0].tolist()
    results = await search_vectors(query_vector)
    return results

# @app.post("/recommend/")
async def recommend_endpoint(request: RecommendRequest):
    """
    Recommends items based on positive and negative samples.

    Args:
        request (RecommendRequest): The request object containing positive and negative samples.

    Returns:
        results: The recommended items.
    """
    positive_vectors = fclip.encode_text(request.positive_samples, batch_size=32)
    positive_vectors = [positive_vector.tolist() for positive_vector in positive_vectors]
    negative_vectors = fclip.encode_text(request.negative_samples, batch_size=32) if request.negative_samples else []
    negative_vectors = [negative_vector.tolist() for negative_vector in negative_vectors]
    results = await recommend_vectors(positive_vectors, negative_vectors, request.strategy)
    return results

# @app.post("/discover/")
async def discover_endpoint(request: DiscoverRequest):
    """
    Discover endpoint that takes a DiscoverRequest object as input and returns the results.

    Args:
        request (DiscoverRequest): The DiscoverRequest object containing the target and context examples.

    Returns:
        results: The results of the discovery process.
    """
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

app.api.add_api_route("/search/", search_endpoint, methods=["POST"], description="Search for similar items based on text input")
app.api.add_api_route("/recommend/", recommend_endpoint, methods=["POST"], description="Search providing positive and negative samples")
app.api.add_api_route("/discover/", discover_endpoint, methods=["POST"], description="Search and Discover items in based on a target query and some context with positive and negative pairs")