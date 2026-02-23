from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 평가기준</h1><br><p id='40' data-category='list' style='font-size:16px'>1) "
 '씹어먹는 기능의 장해는 윗니(상악치아)와 아랫니(하<br>악치아)의 맞물림(교합), 배열상태 및 아래턱의 개구<br>(입을 벌림)운동, '
 '삼킴(연하)운동 등에 따라 종합적<br>으로 판단하여 결정한다.<br>2) “씹어먹는 기능에 심한 장해를 남긴 때”라 함은 심<br>한 '
 '개구(입을 벌림)운동 제한이나 저작(씹기)운동 제<br>한으로 물이나 이에 준하는 음료 이외는 섭취하지 못<br>하는 경우를'),
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
