from langchain_core.documents import Document

chunk = Document(
    page_content=('- 진, 예방적 검사를 위한 비용\n'
 '- 2. 임신, 출산(제왕절개를 포함합니다.), 인공유산과 관련된 비용 및 출산 후 증상 치\n'
 '- 료 비용\n'
 '- 3. 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용\n'
 '- 4. 산후 문제행동, 수유에 따르는 칼슘 부족에 의한 경련 및 기타 임신ㆍ출산과 관련\n'
 '- 된 질병 치료에 대한 비용\n'
 '- 5. 손톱의 절제(며느리발톱의 제거 포함), 잔존유치, 잠복고환,\n'
 '- 배꼽허니아(배꼽부위탈장), 항문낭 제거 등 건강동물에 실시하는 외과수술 및 기타\n'
 '- 검사 또는 손톱깎기 등의 처치비용'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
