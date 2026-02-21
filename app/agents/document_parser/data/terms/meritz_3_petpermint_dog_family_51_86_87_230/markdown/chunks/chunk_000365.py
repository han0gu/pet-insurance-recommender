from langchain_core.documents import Document

chunk = Document(
    page_content=('- 발생일로부터 과거 1년 이내의 예방접종 기록이 있는\n'
 '- 경우에는 보상합니다.)\n'
 ': 파보 바이러스 감염, 디스템퍼 바이러스 감염, 파라\n'
 '인플루엔자 감염, 전염성 간염, 아데노 바이러스 2\n'
 '형 감염, 광견병, 코로나 바이러스 감염, 렙토스피\n'
 '라 감염, 필라리아(심장사상충) 감염, 인플루엔자\n'
 '감염- ③ 상병명을 알 수 없는 상해 또는 질병에 대한 치료\n'
 '- ④ 백신 접종비용 및 기타 질병예방을 위한 검사 또는\n'
 '- 투약·예방 접종비용 및 정기검진, 예방적 검사를\n'
 '- 위한 비용\n'
 '- ⑤ 반려동물의 임신·출산, 제왕절개, 인공유산과 관련'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
