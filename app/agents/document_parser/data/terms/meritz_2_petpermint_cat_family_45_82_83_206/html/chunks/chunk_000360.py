from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 반려동물의 나이 및<br>품종이 정정되기 이전에는 「나이 및 품종이 정정되기 전에<br>적용된 보험료율」의 「나이 및 품종이 '
 "정정된 후에 적용해<br>야할 보험료율」에 대한 비율에 따라 보험금을 삭감하여 지<br>급합니다.</p><h1 id='6' "
 "style='font-size:20px'>제15조(재가입)</h1><br><p id='7' data-category='paragraph' "
 "style='font-size:16px'>\uf000 이 특별약관에서 재가입 적용대상 특별약관(이하「재가<br>입 적용대상 특별약관」이라 "
 '합니다)이라'),
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
