from langchain_core.documents import Document

chunk = Document(
    page_content=(". 또한 원인과 질환에 따라 동시에 사용될 수 있습니<br>다.</p><br><p id='13' data-category='list' "
 "style='font-size:20px'>- 검표(+) : 원인이 되는 질환에 대한 질병분류코드<br>- 별표(*) : 원인(검표)으로 "
 "인한 발현증세에 대한 질<br>병분류코드</p><br><p id='14' data-category='paragraph' "
 "style='font-size:20px'>\uf000 지급금과 이자율 관련 용어</p><br><table id='15'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
