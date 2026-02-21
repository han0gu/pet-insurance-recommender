from langchain_core.documents import Document

chunk = Document(
    page_content=('- 리에 대한 태만\n'
 '- 8. 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 유사\n'
 '- 한 목적으로 이용함으로써 발생한 손해\n'
 '- 9. 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의사 자격이 없는 자의 치\n'
 '- 료행위로 인한 비용 및 그로 인하여 가중된 비용\n'
 '- 10. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태\n'
 '- 11. 대한민국 이외의 지역에서 발생한 사고 및 손해\n'
 '- 부 가 설 명\n'
 '| ∙ 핵연료물질 : 사용된 | 연료를 포함합니다. |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000559',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
