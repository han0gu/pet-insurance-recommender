from langchain_core.documents import Document

chunk = Document(
    page_content=(". 약간의 추상(추한 모습)</h1><br><h1 id='66' style='font-size:20px'>1) 얼굴</h1><br><p "
 "id='67' data-category='paragraph' style='font-size:20px'>가) 손바닥 크기 1/4 이상의 "
 '추상(추한 모습)<br>나) 길이 5cm 이상의 추상반흔(추한 모습의 흉터)<br>다) 지름 2cm 이상의 조직함몰<br>라) 코의 '
 "1/4이상 결손</p><br><h1 id='68' style='font-size:20px'>2) 머리</h1><br><p id='69'"),
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
