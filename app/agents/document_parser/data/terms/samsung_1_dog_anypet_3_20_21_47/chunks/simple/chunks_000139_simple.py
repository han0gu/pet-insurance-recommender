from langchain_core.documents import Document

chunk = Document(
    page_content=('14. 수렵, 투견, 경주등과 수색, 마약탐지, 경계등의 특수목적으로 업무수행 및 훈련 중에 발생한 배 상책임 15. 가입동물의 소음, '
 '냄새, 털날림으로 인하여 발생한 배상책임 16. 가입동물이 질병을 전염시켜 발생한 배상책임 17. 피보험자의 피용인이 피보험자의 업무에 '
 '종사 중에 입은 신체의 장해(상해, 질병 및 그로 인한 사망을 말합니다)에 기인하는 배상책임 18. 동물보호법 시행규칙 제1조의 2에 '
 '따른 맹견의 경우 동법 시행규칙 제12조 제2항에 따라 목줄 과 입마개를 하지 않아 발생한 손해에 대한 배상책임'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000139',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
