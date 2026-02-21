from langchain_core.documents import Document

chunk = Document(
    page_content=(". 반려동물 비용손해 관련 특별약관</h1><p id='72' data-category='paragraph' "
 "style='font-size:20px'>반려동물 비용손해 관련 특별약관 일반조항</p><h1 id='73' "
 "style='font-size:18px'>제1조(목적)</h1><br><p id='74' data-category='paragraph' "
 "style='font-size:16px'>이 특별약관은 계약자와 회사 사이에 피보험자 소유의 보험<br>증권에 기재된 반려동물의 질병 "
 '또는 상해로 인한 손해를<br>보장하기 위하여'),
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
