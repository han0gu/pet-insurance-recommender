from langchain_core.documents import Document

chunk = Document(
    page_content=("상해에 대한 위험을 보장하기 위하여 체<br>결됩니다.</p><h1 id='3' style='font-size:18px'>제2조(용어의 "
 "정의)</h1><br><p id='4' data-category='paragraph' style='font-size:18px'>이 "
 '계약에서 사용되는 용어의 정의는, 이 계약의 다른 조항<br>에서 달리 정의되지 않는 한 다음과 같습니다.</p><br><h1 '
 "id='5' style='font-size:18px'>\uf000 계약 관련 용어</h1><br><table id='6'"),
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
