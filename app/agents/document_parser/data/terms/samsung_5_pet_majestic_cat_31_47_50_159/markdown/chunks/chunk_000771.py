from langchain_core.documents import Document

chunk = Document(
    page_content=('감소비를 압박률로 정한다.\n'
 '다) 척추(등뼈)의 기형장해는 「산업재해보상보험법 시행규칙」 상 경추부, 흉추\n'
 '부, 요추부로 구분하여 각각을 하나의 운동단위로 보며, 하나의 운동단위- \n'
 '- 140 -내에서 여러 개의 척추체(척추뼈 몸통)에 압박골절이 발생한 경우에는 각\n'
 '척추체(척추뼈 몸통)의 압박률을 합산하고, 두 개 이상의 운동단위에서 장\n'
 '해가 발생한 경우에는 그 중 가장 높은 지급률을 적용한다.- 3) 척추(등뼈)의 장해는 퇴행성 기왕증 병변과 사고가 그 증상을 악화시킨 '
 '부분만\n'
 '- 큼, 즉 이 사고와의 관여도를 산정하여 평가한다.'),
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
