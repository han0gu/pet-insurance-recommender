from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 진단시점에 이미 극심한 치매 또는 심<br>한 치매로 진행된 경우에는 6개월간 지속적인 치<br>료 후 평가한다.<br>다) '
 '치매의 장해평가는 전문의(정신건강의학과, 신경<br>과)에 의한 임상치매척도(한국판 Expanded<br>Clinical Dementia '
 "Rating) 검사결과에 따른다.</p><br><p id='49' data-category='list'></p><h1 id='50' "
 "style='font-size:20px'>4) 뇌전증</h1><br><p id='51' data-category='list'"),
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
