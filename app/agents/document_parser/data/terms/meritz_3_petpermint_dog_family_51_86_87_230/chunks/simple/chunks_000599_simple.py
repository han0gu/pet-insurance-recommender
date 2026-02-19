from langchain_core.documents import Document

chunk = Document(
    page_content=('에 대한 응급처치, 긴급호송 또는 그 밖의 긴급조치를 포함합니다)\n'
 '② 제3자로부터 손해의 배상을 받을 수 있는 경우에는 그 권리의 보전 또는 행사를 위한 필요한 조치를 취하는 일 ③ 손해배상책임의 전부 '
 '또는 일부에 관하여 지급(변제), 승인 또는 화해를 하거나 소송, 중재 또는 조정을 제 기하거나 신청하고자 할 경우에는 미리 회사의 '
 '동의를 받는 일\n'
 '\uf000 계약자 또는 피보험자가 정당한 이유없이 제1항의 의무 를 이행하지 않았을 때에는 그 손해에서 다음의 금액을 뺍 니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 179},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000599',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
