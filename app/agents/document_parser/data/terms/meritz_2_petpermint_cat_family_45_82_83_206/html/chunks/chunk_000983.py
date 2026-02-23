from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제2천추 이하의 천골 및 미골은 체간골<br>의 장해로 평가한다.</p><br><p id='81' "
 "data-category='paragraph' style='font-size:16px'>2) 척추(등뼈)의 기형장해는 척추체(척추뼈 "
 '몸통을 말하며,<br>횡돌기 및 극돌기는 제외한다'),
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
