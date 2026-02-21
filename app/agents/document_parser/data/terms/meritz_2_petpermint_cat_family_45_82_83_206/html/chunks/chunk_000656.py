from langchain_core.documents import Document

chunk = Document(
    page_content=("정의와 장소)</h1><br><p id='41' data-category='paragraph' "
 "style='font-size:16px'>이 계약에 있어서 「입원」이라 함은 수의사가 상해 또는<br>질병의 치료가 필요하다고 인정한 "
 '경우로서, 자택 등에서의<br>치료가 곤란하여 동물병원에 입실하여 수의사의 관리 하에<br>치료에 전념하는 것을 말합니다.</p><h1 '
 "id='42' style='font-size:16px'>제5조(특별약관의 소멸)</h1><br><p id='43' "
 "data-category='paragraph'"),
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
