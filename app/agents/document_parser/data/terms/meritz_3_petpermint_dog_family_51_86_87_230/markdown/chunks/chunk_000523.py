from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기재된 배우자(이하「배우자」라 합니다)\n'
 '- ③ 피보험자 본인 또는 배우자와 생계를 같이 하는 동거\n'
 '- 친족 및 별거 중인 미혼자녀\n'
 '# 【민법 제777조(친족의 범위)】친족관계로 인한 법률상 효력은 이 법 또는 다른 법률에\n'
 '특별한 규정이 없는 한 다음 각호에 해당하는 자에 미친\n'
 '다.- 1. 8촌이내의 혈족\n'
 '- 2. 4촌이내의 인척\n'
 '- 3. 배우자\n'
 '\uf000 제2항에서 피보험자 본인과 본인 이외의 피보험자와의'),
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
