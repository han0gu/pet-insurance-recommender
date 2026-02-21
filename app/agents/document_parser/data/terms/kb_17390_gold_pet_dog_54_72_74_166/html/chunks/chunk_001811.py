from langchain_core.documents import Document

chunk = Document(
    page_content=('장애</td></tr><tr><td>삼첨판폐쇄부전</td></tr><tr><td>심근증</td></tr><tr><td>심낭수종</td></tr><tr><td>심내막염</td></tr><tr><td>심방중격결손</td></tr><tr><td>심부전 '
 '심비대</td></tr><tr><td>심실중격결손</td></tr><tr><td>심정지</td></tr><tr><td>우대동맥궁 '
 '잔존</td></tr><tr><td>이첨판폐쇄부전</td></tr><tr><td>폐동맥협착증</td></tr><tr><td>폐성고혈압 '
 '혈전 /'),
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
