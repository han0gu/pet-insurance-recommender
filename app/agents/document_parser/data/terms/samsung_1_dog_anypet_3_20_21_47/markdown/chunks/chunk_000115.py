from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항의 의무보험은 피보험자가 법률에 의하여 의무적으로 가입하여야 하는 보험으로서 공제계약\n'
 '- 을 포함합니다.\n'
 '- 26 -당신에게 좋은보험 삼성화재③ 피보험자가 의무보험에 가입하여야 함에도 불구하고 가입하지 않은 경우에는 그가 가입했더라면\n'
 '의무보험에서 보상했을 금액을 제1항의 "의무보험에서 보상하는 금액"으로 봅니다.# 제6조(보험금의 분담)① 이 계약에서 보장하는 위험과 '
 '같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)이 있을 경'),
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
