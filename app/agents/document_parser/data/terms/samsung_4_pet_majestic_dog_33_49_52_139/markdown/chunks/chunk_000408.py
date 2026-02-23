from langchain_core.documents import Document

chunk = Document(
    page_content=('단 또는 치료를 받고 있었음을 증명할 수 있는 문서화된 기록 또는 증거를 진단확정\n'
 '의 기초로 할 수 있습니다.- \n'
 '# <관련법규>- [감염병의 예방 및 관리에 관한 법률 시행규칙 [별표2] 감염병환자등의 진단기준]\n'
 '∙ 감염병환자 : 해당 감염병에 부합되는 임상적 특징을 나타내면서 특정 검사방법으로 감염병 병원\n'
 '체가 확인된 사람\n'
 '∙ 의사환자 : 임상적 특징 및 역학적 연관성을 고려할 때 해당 감염병이 의심되나 감염병 병원체가\n'
 '확인되지 않은 사람\n'
 '∙ 병원체보유자 : 임상증상을 나타내지 않으나 감염병병원체가 확인된 사람-'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
