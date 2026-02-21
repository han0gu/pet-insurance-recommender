from langchain_core.documents import Document

chunk = Document(
    page_content=("이용하여 반려동물의 횡단면상의 영상을 획득하여<br>진단에 이용하는 검사</p><h1 id='51' "
 "style='font-size:20px'>【내시경】</h1><br><p id='52' data-category='paragraph' "
 "style='font-size:16px'>내장장기 또는 체강(體腔) 내부를 직접 볼 수 있게 만든<br>의료기구</p><h1 "
 "id='53' style='font-size:20px'>제5조(특별약관의 소멸)</h1><br><p id='54' "
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
