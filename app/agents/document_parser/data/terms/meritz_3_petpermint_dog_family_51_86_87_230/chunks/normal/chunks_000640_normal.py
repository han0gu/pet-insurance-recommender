from langchain_core.documents import Document

chunk = Document(
    page_content=('【심신상실】\n'
 '정신병, 정신박약, 심한 의식장애 등의 심신장애로 인하 여 사물 변별 능력 또는 의사 결정 능력이 없는 상태를 말합니다.\n'
 '⑩ 피보험자의 지시에 따른 배상책임 ⑪ 벌과금 및 징벌적 손해에 대한 배상책임 ⑫ 피보험자와 세대를 같이하는 친족에 대한 배상책임 ⑬ '
 '범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 목적으로 이용하는 중에 발생한 손해 에 대한 배상책임 ⑭ 가입 '
 '반려견의 소음, 냄새, 털날림으로 인하여 발생한 배상책임 ⑮ 가입 반려견이 질병을 전염시켜 발생한 배상책임\n'
 '제3조(특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 188},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000640',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
