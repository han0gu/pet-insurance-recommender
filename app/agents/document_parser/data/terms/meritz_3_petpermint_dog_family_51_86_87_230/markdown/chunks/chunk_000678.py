from langchain_core.documents import Document

chunk = Document(
    page_content=('- 삽입한 경우\n'
 '- 바) 요도괄약근 등의 기능장해로 영구적으로 인공요\n'
 '- 도괄약근을 설치한 경우\n'
 '5) “흉복부장기 또는 비뇨생식기 기능에 약간의 장해를\n'
 '남긴 때”라 함은 아래의 경우 중 하나에 해당하는\n'
 '때를 말한다.- 가) 방광의 용량이 50cc 이하로 위축되었거나 요도협\n'
 '- 착, 배뇨기능 상실로 영구적인 간헐적 인공요도\n'
 '- 가 필요한 때\n'
 '- 나) 음경의 1/2 이상이 결손되었거나 질구 협착으로\n'
 '- 성생활이 불가능한 때\n'
 '- 다) 폐질환 또는 폐 부분절제술 후 일상생활에서 호\n'
 '- 흡곤란으로 지속적인 산소치료가 필요하며, 폐기'),
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
