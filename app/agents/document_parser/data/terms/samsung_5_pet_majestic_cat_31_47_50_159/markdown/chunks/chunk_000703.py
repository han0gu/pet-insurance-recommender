from langchain_core.documents import Document

chunk = Document(
    page_content=('| 14 | 구강, 치아, 혀, 악하선(턱밑샘), 이하선(귀밑샘) 및 설하선(혀밑샘) |\n'
 '| 15 | 귀[외이(바깥 귀), 고막, 중이(가운데귀), 내이(속귀), 청신경 및 유양돌기(꼭지돌기)포함] |\n'
 '| 16 | 안구 및 안구부속기[안검(눈꺼풀), 결막, 누기(눈물샘), 안근 및 안와내조직 포함] |\n'
 '| 17 | 신장 |\n'
 '| 18 | 부신 |\n'
 '| 19 | 요관, 방광 및 요도 |\n'
 '| 20 | 음경 |\n'
 '| 21 | 질 및 외음부 |\n'
 '| 22 | 전립선 |\n'
 '| 23 | 유방(유선 포함) |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
