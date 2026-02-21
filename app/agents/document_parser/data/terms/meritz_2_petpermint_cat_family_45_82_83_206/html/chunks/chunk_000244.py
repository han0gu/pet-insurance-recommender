from langchain_core.documents import Document

chunk = Document(
    page_content=("제1항)</p><h1 id='36' style='font-size:18px'>【 민법 제2조(신의성실) 제1항 】</h1><br><p "
 "id='37' data-category='paragraph' style='font-size:18px'>① 권리의 행사와 의무의 이행은 "
 "신의에 좇아 성실히 하<br>여야 한다.</p><footer id='38' "
 "style='font-size:14px'>79</footer><p id='39' data-category='paragraph' "
 "style='font-size:16px'>\uf000 회사는 약관의 뜻이"),
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
