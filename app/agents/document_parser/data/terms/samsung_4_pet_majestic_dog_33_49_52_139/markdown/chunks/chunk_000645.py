from langchain_core.documents import Document

chunk = Document(
    page_content=('- 으로 이용함으로써 발생한 손해\n'
 '- 12. 가입동물의 소음, 냄새, 털날림으로 인하여 발생한 배상책임\n'
 '- 13. 가입동물이 질병을 전염시켜 발생한 배상책임\n'
 '- 14. 동물보호법 시행규칙 제1조의 3에 따른 맹견의 경우 동법 시행규칙 제12조 제2항\n'
 '# 에 따라 목줄과 입마개를 하지 않아 발생한 손해에 대한 배상책임# 제5조 (의무보험과의 관계)- ① 회사는 이 특별약관에 의하여 '
 '보상하여야 하는 금액이 의무보험에서 보상하는 금액을\n'
 '- 초과할 때에 한하여 그 초과액만을 보상합니다. 다만, 의무보험이 다수인 경우에는 제'),
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
