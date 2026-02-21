from langchain_core.documents import Document

chunk = Document(
    page_content=('- 권리를 가진 사람에게 부담하는 손해에 대한 배상책임\n'
 '- 5. 피보험자와 타인 간에 손해배상에 관한 약정이 있는 경우, 그 약정에 의하여 가중된 배상책임\n'
 '- 6. 핵연료물질 또는 핵연료 물질에 의하여 오염된 물질의 방사성, 폭발성 그 밖의 유해한 특성 또\n'
 '- 는 이들의 특성에 의한 사고로 생긴 손해에 대한 배상책임\n'
 '- 7. 위 제6호 이외의 방사선을 쬐는 것 또는 방사능 오염으로 인한 손해\n'
 '- 8. 티끌, 먼지, 석면, 분진 또는 소음으로 생긴 손해에 대한 배상책임\n'
 '- 9. 전자파, 전자장(EMF)으로 생긴 손해에 대한 배상책임'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
