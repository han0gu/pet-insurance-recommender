from langchain_core.documents import Document

chunk = Document(
    page_content=("이내에 제1항의 절차를 이행할 수 있습니다.</p><br><h1 id='51' "
 "style='font-size:20px'>【용어풀이】</h1><br><p id='52' data-category='list' "
 "style='font-size:20px'>1"),
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
