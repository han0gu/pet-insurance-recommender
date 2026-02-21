from langchain_core.documents import Document

chunk = Document(
    page_content=('- 입었을 경우에 그 재물에 대하여 정당한 권리를 가진\n'
 '- 사람에게 부담하는 배상책임\n'
 '- ⑨ 피보험자의 심신상실로 인한 배상책임\n'
 '187# 【심신상실】정신병, 정신박약, 심한 의식장애 등의 심신장애로 인하\n'
 '여 사물 변별 능력 또는 의사 결정 능력이 없는 상태를\n'
 '말합니다.- ⑩ 피보험자의 지시에 따른 배상책임\n'
 '- ⑪ 벌과금 및 징벌적 손해에 대한 배상책임\n'
 '- ⑫ 피보험자와 세대를 같이하는 친족에 대한 배상책임\n'
 '- ⑬ 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험\n'
 '- 및 이와 유사한 목적으로 이용하는 중에 발생한 손해'),
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
