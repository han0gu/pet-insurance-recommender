from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약일 보장개시일(책임개시일)\n'
 '◄───── 30일주2) ─────►\n'
 '2022년 8월 1일 2022년 8월 31일주1) 상해를 직접적인 원인으로 치료를 받은 경우에는 보장개시일(책임개시일)은 보험계약일로 '
 '합\n'
 '니다.\n'
 '주2) 암, 백내장, 녹내장, 심장질환, 신장질환, 방광질환 및 각종 결석의 경우 90일- ④ 제1항 내지 제3항에도 불구하고 '
 '보험계약일부터 그 날을 포함하여 1년 이내에 발생한\n'
 '- 슬관절탈구, 고관절탈구, 슬관절형성부전, 고관절형성부전 또는 기타 이들과 유사한'),
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
