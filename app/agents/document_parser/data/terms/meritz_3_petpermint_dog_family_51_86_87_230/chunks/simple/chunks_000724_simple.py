from langchain_core.documents import Document

chunk = Document(
    page_content=('나. 장해의 평가기준\n'
 '1) 씹어먹는 기능의 장해는 윗니(상악치아)와 아랫니(하 악치아)의 맞물림(교합), 배열상태 및 아래턱의 개구 (입을 벌림)운동, '
 '삼킴(연하)운동 등에 따라 종합적 으로 판단하여 결정한다. 2) “씹어먹는 기능에 심한 장해를 남긴 때”라 함은 심 한 개구(입을 '
 '벌림)운동 제한이나 저작(씹기)운동 제 한으로 물이나 이에 준하는 음료 이외는 섭취하지 못 하는 경우를 말한다. 3) “씹어먹는 기능에 '
 '뚜렷한 장해를 남긴 때”라 함은 아래의 경우 중 하나 이상에 해당되는 때를 말한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 207},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000724',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
