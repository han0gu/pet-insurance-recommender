from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해판정기준</h1><h1 id='35' style='font-size:20px'>1) 신경계</h1><br><p id='36' "
 "data-category='list' style='font-size:20px'>가) “신경계에 장해를 남긴 때”라 함은 뇌, "
 '척수<br>및 말초신경계 손상으로 “<붙임>일상생활 기본<br>동작(ADLs) 제한 장해평가표”의 5가지 기본동작<br>중 하나 이상의 '
 '동작이 제한되었을 때를 말한다.<br>나) 위 가)의 경우 “<붙임>일상생활 기본동작(ADLs) 제<br>한 장해평가표”상 지급률이 '
 '10%'),
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
