from langchain_core.documents import Document

chunk = Document(
    page_content=('- 으로 이용함으로써 발생한 손해\n'
 '- 12. 가입동물의 소음, 냄새, 털날림으로 인하여 발생한 배상책임\n'
 '# 에 따라 목줄과 입마개를 하지 않아 발생한 손해에 대한 배상책임# 제5조 (의무보험과의 관계)- ① 회사는 이 특별약관에 의하여 '
 '보상하여야 하는 금액이 의무보험에서 보상하는 금액을\n'
 '- 초과할 때에 한하여 그 초과액만을 보상합니다. 다만, 의무보험이 다수인 경우에는 제\n'
 '- 10조(보험금의 분담)를 따릅니다.\n'
 '- 제1항의 의무보험은 피보험자가 법률에 의하여 의무적으로 가입하여야 하는 보험으로\n'
 '- 서 공제계약을 포함합니다.'),
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
