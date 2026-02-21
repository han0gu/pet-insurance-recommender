from langchain_core.documents import Document

chunk = Document(
    page_content=('변경지정한 다음의 자(이하"지정대리청구인"이라 합니다)가 제6조(보험금의 청\n'
 '구)에 정한 구비서류 및 특별한 사정이 있음을 증명하는 서류를 제출하고 회사의\n'
 '승낙을 얻어 이 특별약관의 보험금 수익자의 대리인으로서 이 특별약관의 보험금\n'
 '을 청구할 수 있습니다.\n'
 '1. 피보험자의 가족관계등록부상 또는 주민등록상의 배우자- 2. 피보험자의 3촌 이내의 친족\n'
 '- \uf000 제1항의 규정에 의하여 회사가 이 특별약관의 보험금을\n'
 '지정대리청구인에게 지급한 경우에는 그 이후 이 특별약관의 보험금 청구를 받더라도 회사는 이를 지급하'),
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
