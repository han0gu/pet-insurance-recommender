from langchain_core.documents import Document

chunk = Document(
    page_content=('- 9. 전자파, 전자장(EMF)으로 생긴 손해에 대한 배상책임\n'
 '- 10. 벌과금 및 징벌적 손해에 대한 배상책임\n'
 '- 11. 피보험자의 심신상실에 기인하는 배상책임\n'
 '- 12. 피보험자의 지시에 따른 배상책임\n'
 '- 13. 피보험자의 불법행위 또는 폭력행위에 기인하는 배상책임\n'
 '- 25 -당신에게 좋은보험 삼성화재- 14. 수렵, 투견, 경주등과 수색, 마약탐지, 경계등의 특수목적으로 업무수행 및 훈련 중에 '
 '발생한 배\n'
 '- 상책임\n'
 '- 15. 가입동물의 소음, 냄새, 털날림으로 인하여 발생한 배상책임'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000109',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
