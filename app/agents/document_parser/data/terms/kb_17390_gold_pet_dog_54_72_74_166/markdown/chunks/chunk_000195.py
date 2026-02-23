from langchain_core.documents import Document

chunk = Document(
    page_content=('드립니다.\n'
 '1. 지정대리청구인 변경신청서(회사양식)제출하고 지정대리청구인을 변경 지정할 수 있- 2. 지정대리청구인의 주민등록등본, 가족관계등록부 '
 '(기본증명서 등)\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인\n'
 '- 이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰'),
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
