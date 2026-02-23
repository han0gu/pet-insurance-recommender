from langchain_core.documents import Document

chunk = Document(
    page_content=('습니다.- 1. 피보험자의 피용인이 피보험자의 업무에 종사 중에 입은 신체의 피해로 인한 배상\n'
 '- 책임\n'
 '- 2. 피보험자와 타인간에 손해배상에 관한 약정이 있는 경우 그 약정에 의하여 가중된\n'
 '- 배상책임. 그러나 약정이 없었더라도 법률규정에 의하여 피보험자가 부담하게 될\n'
 '- 배상책임은 보상합니다.\n'
 '- 3. 피보험자와 세대를 같이하는 친족에 대한 배상책임\n'
 '- 4. 피보험자가 소유, 사용 또는 관리하는 재물이 손해를 입었을 경우에 그 재물에 대\n'
 '- 하여 정당한 권리를 가진 사람에게 부담하는 손해에 대한 배상책임.'),
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
