from langchain_core.documents import Document

chunk = Document(
    page_content=('. "동물병원"이란 동물진료업을 하는 장소로서 제17조에 따른 신고를 한 진료</td></tr></tbody></table><br><p '
 "id='161' data-category='paragraph' style='font-size:14px'>기관을 말한다.<br>\uf000 "
 "제1항 제4호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 내용에 따라</p><br><h1 id='162' "
 "style='font-size:14px'>국내의 동물병원에서 수의사에 의해 발급한 것이어야 합니다.</h1><br><table "
 "id='163'"),
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
