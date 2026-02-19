from langchain_core.documents import Document

chunk = Document(
    page_content=('. 5. 피보험자의 심신상실로 인한 배상책임 6. 피보험자의 지시에 따른 배상책임 7. 피보험자의 불법행위 또는 폭력행위로 인한 배상책임 '
 '8. 티끌, 먼지, 석면, 분진 또는 소음으로 생긴 손해에 대한 배상책임 9. 전자파, 전자장(EMF)으로 생긴 손해에 대한 배상책임 '
 '10. 벌과금 및 징벌적 손해에 대한 배상책임 11. 반려견을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사한 '
 '목적 으로 이용함으로써 발생한 손해 12. 가입동물의 소음, 냄새, 털날림으로 인하여 발생한 배상책임'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000567',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
