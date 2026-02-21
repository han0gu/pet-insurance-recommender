from langchain_core.documents import Document

chunk = Document(
    page_content=('- 할 보험금을 결정하고 지급할 보험금이 결정되면 7일 이내에 이를 지급합니다.\n'
 '- \uf000 제1항에 의한 지급할 보험금이 결정되기 전이라도 피보험자의 청구가 있을 때에\n'
 '- 는 회사가 추정한 보험금의 50% 상당액을 가지급보험금으로 지급합니다.\n'
 '- \uf000 회사는 제1항의 지급보험금이 결정된 후 7일(이하 "지급기일"이라 합니다)이 지\n'
 '- 나도록 보험금을 지급하지 않았을 때에는 지급기일의 다음날부터 지급일까지의\n'
 '- 기간에 대하여 "보험금을 지급할 때의 적립이율 계산(【별표2】참조)"에서 정한'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
