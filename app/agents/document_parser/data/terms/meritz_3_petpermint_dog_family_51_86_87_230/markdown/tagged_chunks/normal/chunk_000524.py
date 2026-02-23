from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 배우자\n'
 '\uf000 제2항에서 피보험자 본인과 본인 이외의 피보험자와의\n'
 '관계는 사고발생 당시의 관계를 말합니다.# \uf000 1사고당 보상하는 손해의 범위는 아래와 같습니다.- ① 피보험자가 피해자에게 '
 '지급할 책임을 지는 법률상의\n'
 '- 손해배상금\n'
 '- ② 계약자 또는 피보험자가 지출한 아래의 비용\n'
 '- ㉠ 피보험자가「배상책임 관련 특별약관 일반조항」제8\n'
 '- 조(손해방지의무)의 제1항 제1호의 손해의 방지 또\n'
 '- 는 경감을 위하여 지출한 필요 또는 유익하였던 비\n'
 '- 용\n'
 '- ㉡ 피보험자가「배상책임 관련 특별약관 일반조항」제8'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000524',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
