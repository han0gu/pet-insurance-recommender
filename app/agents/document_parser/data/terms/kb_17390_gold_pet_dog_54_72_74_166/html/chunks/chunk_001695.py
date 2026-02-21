from langchain_core.documents import Document

chunk = Document(
    page_content=('신체부위를 복합적으로 침범한 탈구,염좌 및 긴장</td><td>T03.8주)</td></tr><tr><td>\u3000목과 함께 머리를 '
 '침범한 으깸손상</td><td>T04.0</td></tr><tr><td>\u3000기타 신체부위를 복합적으로 침범한 '
 '으깸손상</td><td>T04.8주)</td></tr><tr><td>화상 및 부식(화학약품 등에 의한 피부손상) '
 '중,</td><td>\u3000</td></tr><tr><td>\u3000머리 및 목의 화상 및 부식 (화학약품 등에 의한 '
 '피부손상)</td><td>T20</td></tr><tr><td>\u3000눈 및 부속기에'),
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
