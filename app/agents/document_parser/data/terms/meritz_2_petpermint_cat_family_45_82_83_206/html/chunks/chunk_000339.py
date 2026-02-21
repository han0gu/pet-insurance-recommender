from langchain_core.documents import Document

chunk = Document(
    page_content=("1개월 이내)에 계약을 취소할 수 있습니<br>다.</p><footer id='80' "
 "style='font-size:14px'>94</footer><p id='81' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제1항에 따라 계약이 취소된 경우에는 회사는 이미 납입<br>한 보험료를 계약자에게 "
 "돌려 드립니다.</p><h1 id='82' style='font-size:16px'>제11조(보험계약의 성립)</h1><br><p "
 "id='83' data-category='paragraph'"),
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
