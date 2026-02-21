from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>151</footer><p id='45' data-category='paragraph' "
 "style='font-size:20px'>제4조(MRI,CT 및 내시경처치의 정의)</p><br><p id='46' "
 "data-category='paragraph' style='font-size:16px'>\uf000 이 특별약관에 있어서 MRI,CT 및 "
 '내시경처치라 함은 자<br>기공명영상(MRI), 전산화단층촬영(CT) 및 내시경처치를 말<br>합니다.<br>\uf000 제1항의 '
 '자기공명영상(MRI)이라 함은'),
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
