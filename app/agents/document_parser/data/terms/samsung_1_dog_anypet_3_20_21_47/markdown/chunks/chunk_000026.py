from langchain_core.documents import Document

chunk = Document(
    page_content=('드립니다. 또한, 지급할 보험금이 결정되기 전이라도 피보험자의 청구가 있을 때에는 회사가 추정\n'
 '한 보험금의 50% 상당액을 가지급보험금으로 지급합니다.【가지급보험금】 보험금이 지급기한 내에 지급되지 못할 것으로 판단되는 경우 회사가 '
 '예상되는 보험금\n'
 '의 일부를 먼저 지급하는 제도로 피보험자가 필요로 하는 비용을 보전해 주기 위해 회사가 먼저 지급하는\n'
 "임시 교부금을 말합니다.회사는 제1항의 지급보험금이 결정된 후 7일(이하 '지급기일'이라 합니다)이 지나도록 보험금을 지급"),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
