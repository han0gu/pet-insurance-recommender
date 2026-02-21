from langchain_core.documents import Document

chunk = Document(
    page_content=('- 및 이와 유사한 목적으로 이용하는 중에 발생한 손해\n'
 '- 에 대한 배상책임\n'
 '- ⑭ 가입 반려견의 소음, 냄새, 털날림으로 인하여 발생한\n'
 '- 배상책임\n'
 '- ⑮ 가입 반려견이 질병을 전염시켜 발생한 배상책임\n'
 '# 제3조(특별약관의 소멸)이 특별약관에서 정한 보상하는 손해가 더 이상 발생할 수\n'
 '없는 경우에는 이 특별약관은 그 때부터 소멸되며, 이 경우\n'
 '회사는「보험료 및 해약환급금 산출방법서」에서 정한 이\n'
 '특별약관의 그 때까지 적립한 계약자적립액 및 미경과보험'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000531',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
