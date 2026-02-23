from langchain_core.documents import Document

chunk = Document(
    page_content=('- 수 있는 증상을 포함합니다. 다만, 보험기간 중 최초로 발견된 경우에는 해당 보험\n'
 '- 기간에 한하여 보상합니다.)\n'
 '- 2. 다음 정한 질병 및 이에 기인하는 질병(다만, 질병의 발생일로부터 과거 1년 이내의\n'
 '- 동물병원 예방접종 기록이 있는 경우에는 보상합니다.)\n'
 '- 4 -- : 파보 바이러스 감염, 디스템퍼 바이러스 감염, 파라 인플루엔자 감염, 전염성 간염,\n'
 '- 아데노 바이러스 2형 감염, 광견병, 코로나 바이러스 감염, 렙토스피라 감염, 필라리'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
