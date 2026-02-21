from langchain_core.documents import Document

chunk = Document(
    page_content=("마지막 날이 영업일이 아닌</p><footer id='56' style='font-size:14px'>72</footer><p "
 "id='57' data-category='paragraph' style='font-size:16px'>때에는 최고(독촉)기간은 그 다음 "
 '날까지로 합니다)으로 정<br>하여 아래 사항에 대하여 서면(등기우편 등), 전화(음성녹<br>음) 또는 전자문서 등으로 알려드립니다'),
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
