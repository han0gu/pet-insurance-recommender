from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에는 적용하지 않습니다.\n'
 '- 8. 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또는 급수 등 기본적인 관리에 대한 태만\n'
 '- 9. 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조, 실험 및 이와 유사한 목적으로 이용함으로써\n'
 '- 발생한 손해\n'
 '- 10. 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의사 자격이 없는 자의 치료행위로 인한 비\n'
 '- 용 및 그로 인하여 가중된 비용\n'
 '- 11. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태\n'
 '- 12. 대한민국 이외의 지역에서 발생한 사고 및 손해'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
