from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나) 척추체(척추뼈 몸통) 한 개의 압박률이 40%이상인 경우 또는 한 운동단위\n'
 '- 내에 두 개 이상 척추체(척추뼈 몸통)의 압박골절로 각 척추체(척추뼈 몸\n'
 '- 통)의 압박률의 합이 60% 이상일 때\n'
 '# 11) 약간의 기형이란 다음 중 어느 하나에 해당하는 경우를 말한다.- 가) 1개 이상의 척추(등뼈)의 골절 또는 탈구로 경도(가벼운 '
 '정도)의 척추전만증\n'
 '- (척추가 앞으로 휘어지는 증상), 척추후만증(척추가 뒤로 휘어지는 증상)\n'
 '- 또는 척추측만증(척추가 옆으로 휘어지는 증상) 변형이 있을 때'),
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
