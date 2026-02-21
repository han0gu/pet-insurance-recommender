from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 질병 및 상해\n'
 '- ⑦ 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또\n'
 '- 는 급수 등 기본적인 관리에 대한 태만\n'
 '- ⑧ 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조,\n'
 '- 투견, 실험 및 이와 유사한 목적으로 이용함으로써 발\n'
 '- 생한 손해\n'
 '- ⑨ 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의\n'
 '- 사 자격이 없는 자의 치료행위로 인한 비용 및 그로\n'
 '- 인하여 가중된 비용\n'
 '- ⑩ 국가 및 지방자치단체의 명령 또는 법률에 의한 살처\n'
 '- 분 또는 이와 유사한 사태'),
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
