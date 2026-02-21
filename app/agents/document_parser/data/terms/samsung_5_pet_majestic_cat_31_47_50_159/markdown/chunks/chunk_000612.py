from langchain_core.documents import Document

chunk = Document(
    page_content=('6. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류- \n'
 '② 제1항 제4호의 사고증명서는 수의사법 제2조(정의)에서 규정한 동물병원에서 수의사\n'
 '가 발급한 것이어야 합니다.<관련법규># [수의사법 제2조(정의)]- 1. "수의사"란 수의업무를 담당하는 사람으로서 농림축산식품부장관의 '
 '면허를 받은 사람을 말한다.\n'
 '- 2. "동물"이란 소, 말, 돼지, 양, 개, 토끼, 고양이, 조류(鳥類), 꿀벌, 수생동물(水生動物), 그 밖에 대\n'
 '- 통령령으로 정하는 동물을 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
