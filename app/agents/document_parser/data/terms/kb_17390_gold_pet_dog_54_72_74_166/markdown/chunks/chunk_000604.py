from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 증, 전염성 간염, 아데노바이러스2형감염증, 코로나바이러스감염 증, 렙토스피라감염증, 필라리아감염증, 광견병, 인플루엔자 감염, '
 '켄넬코프 | 증, 전염성 간염, 아데노바이러스2형감염증, 코로나바이러스감염 증, 렙토스피라감염증, 필라리아감염증, 광견병, 인플루엔자 '
 '감염, 켄넬코프 |\n'
 '- 114 -- 3. 상병명을 알 수 없는 상해 또는 질병에 대한 치료\n'
 '- 4. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약·예방 접종비용 및 정\n'
 '- 기검진, 예방적 검사를 위한 비용'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
