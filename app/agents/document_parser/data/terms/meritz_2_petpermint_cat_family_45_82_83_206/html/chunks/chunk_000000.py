from langchain_core.documents import Document

chunk = Document(
    page_content=("<header id='0' style='font-size:20px'>제1관 목적 및 용어의 정의</header><h1 id='1' "
 "style='font-size:16px'>제1조(목적)</h1><br><p id='2' data-category='paragraph' "
 "style='font-size:18px'>이 보험계약(이하「계약」이라 합니다)은 보험계약자(이하<br>「계약자」라 합니다)와 "
 "보험회사(이하「회사」라 합니다)<br>사이에 피보험자의 상해에 대한 위험을 보장하기 위하여 체<br>결됩니다.</p><h1 id='3'"),
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
