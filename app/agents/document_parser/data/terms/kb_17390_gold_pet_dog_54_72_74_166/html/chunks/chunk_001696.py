from langchain_core.documents import Document

chunk = Document(
    page_content=('의한 피부손상)</td><td>T20</td></tr><tr><td>\u3000눈 및 부속기에 국한된 화상 및 부식 (화학약품 등에 의한 '
 '피부손상)</td><td>T26</td></tr><tr><td>동상 중,</td><td>\u3000'
 '</td></tr><tr><td>\u3000머리의 표재성 동상</td><td>T33.0</td></tr><tr><td>\u3000목의 '
 '표재성 동상</td><td>T33.1</td></tr><tr><td>\u3000조직괴사를 동반한 머리의 동상 \u3000조직괴사를 동반한 '
 '목의 동상</td><td>T34.0'),
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
